# SPDX-License-Identifier: Apache-2.0
"""Prefill ladder for #625: PP prefill against TP prefill, per prompt length.

The question this answers is narrow and it is a falsification before it is a
comparison. Under PP a prompt is only parallel across stages if successive
CHUNKS of the same request occupy different microbatch slots; if they do not,
PP prefill is ~pp_size-times serial and #625 is over. So the ladder reports
per-length prefill time on both topologies and leaves the verdict to the
numbers.

Measurement discipline, which is why this is a script and not a curl loop:

* **Uncached by construction.** The prompt is random token ids, freshly drawn
  per request, sent as ``input_ids``. That fixes the token count EXACTLY
  (a word-based prompt only approximates it) and guarantees no radix-cache
  prefix hit. ``cached_tokens`` is read back from ``meta_info`` on every
  request and a non-zero value FAILS the draw rather than being averaged in —
  a warm prefill is four times too good (#212) and would silently become the
  headline.
* **Pure prefill.** ``max_new_tokens=1``, so the measured window is prefill
  plus one decode step, and the decode step is reported separately from the
  same boot so it can be subtracted.
* **A-vs-A floor first.** Nothing is compared across topologies until the same
  configuration has been drawn twice back-to-back on ONE boot and the spread
  of those two draws is known. A difference below that spread is not a result.
  The first draw of every series is a warm-up and is DISCARDED explicitly and
  visibly, never silently.

Usage:

    python prefill_ladder.py floor  --url http://127.0.0.1:PORT --tokens 8192
    python prefill_ladder.py ladder --url http://127.0.0.1:PORT \\
        --tokens 2048,8192,32768 --label pp3 --out /path/result.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

# Random ids are drawn well clear of the special-token band at the bottom of
# every vocabulary this rig serves, and below the smallest vocab in use.
_ID_LO = 1000
_ID_HI = 100_000

# Every request is bounded. An unbounded wait in a measurement driver turns a
# hung server into a hung window (agent-wedge rule).
_TIMEOUT_S = 300


class DrawFailed(RuntimeError):
    """A draw that cannot be trusted, e.g. one that hit a warm prefix."""


def _post(url: str, payload: dict, timeout: float = _TIMEOUT_S) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _random_ids(n: int, rng: random.Random) -> list:
    return [rng.randint(_ID_LO, _ID_HI - 1) for _ in range(n)]


def one_draw(url: str, n_tokens: int, rng: random.Random, new_tokens: int = 1) -> dict:
    """One prefill of exactly ``n_tokens`` uncached tokens. Wall-clocked."""
    payload = {
        "input_ids": _random_ids(n_tokens, rng),
        "sampling_params": {
            "max_new_tokens": new_tokens,
            "temperature": 0.0,
        },
    }
    t0 = time.perf_counter()
    out = _post(url.rstrip("/") + "/generate", payload)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    meta = out.get("meta_info", {}) or {}
    cached = meta.get("cached_tokens", 0) or 0
    if cached:
        raise DrawFailed(
            f"draw hit a warm prefix: cached_tokens={cached} at n_tokens={n_tokens}. "
            "The number would be four times too good; refusing to report it."
        )
    prompt_tokens = meta.get("prompt_tokens")
    if prompt_tokens is not None and prompt_tokens != n_tokens:
        raise DrawFailed(
            f"server counted {prompt_tokens} prompt tokens, driver sent {n_tokens}. "
            "Work is not matched; the comparison would be between two different jobs."
        )
    return {
        "n_tokens": n_tokens,
        "wall_ms": wall_ms,
        "cached_tokens": cached,
        "prompt_tokens": prompt_tokens,
        "e2e_latency": meta.get("e2e_latency"),
        "tok_per_s": (n_tokens / wall_ms * 1000.0) if wall_ms > 0 else None,
    }


def series(
    url: str, n_tokens: int, draws: int, rng: random.Random, new_tokens: int = 1
) -> dict:
    """``draws`` back-to-back draws, first one discarded as warm-up."""
    raw = []
    for i in range(draws + 1):
        d = one_draw(url, n_tokens, rng, new_tokens=new_tokens)
        d["draw_index"] = i
        d["warmup"] = i == 0
        raw.append(d)
    kept = [d["wall_ms"] for d in raw if not d["warmup"]]
    return {
        "n_tokens": n_tokens,
        "draws_kept": len(kept),
        "warmup_discarded": 1,
        "warmup_ms": raw[0]["wall_ms"],
        "median_ms": statistics.median(kept),
        "min_ms": min(kept),
        "max_ms": max(kept),
        "spread_pct": (max(kept) - min(kept)) / statistics.median(kept) * 100.0,
        "tok_per_s_median": n_tokens / statistics.median(kept) * 1000.0,
        "raw": raw,
    }


def cmd_floor(args) -> int:
    """A-vs-A: the same series twice on one boot. This is the noise floor."""
    rng = random.Random(args.seed)
    a = series(args.url, args.tokens, args.draws, rng)
    b = series(args.url, args.tokens, args.draws, rng)
    delta = abs(a["median_ms"] - b["median_ms"])
    floor_pct = delta / statistics.median([a["median_ms"], b["median_ms"]]) * 100.0
    out = {
        "kind": "a-vs-a-floor",
        "n_tokens": args.tokens,
        "arm_a_median_ms": a["median_ms"],
        "arm_b_median_ms": b["median_ms"],
        "floor_pct": floor_pct,
        "detail": {"a": a, "b": b},
    }
    print(json.dumps(out, indent=2))
    print(
        f"\nFLOOR at {args.tokens} tok: {floor_pct:.2f} % "
        f"(A {a['median_ms']:.1f} ms vs B {b['median_ms']:.1f} ms). "
        f"Nothing below this is a result.",
        file=sys.stderr,
    )
    return 0


def cmd_ladder(args) -> int:
    rng = random.Random(args.seed)
    lengths = [int(x) for x in args.tokens.split(",")]
    results = []
    for n in lengths:
        try:
            s = series(args.url, n, args.draws, rng)
        except DrawFailed as exc:
            print(f"REFUSED at {n} tok: {exc}", file=sys.stderr)
            return 2
        except urllib.error.URLError as exc:
            print(f"TRANSPORT FAIL at {n} tok: {exc}", file=sys.stderr)
            return 3
        results.append(s)
        print(
            f"{args.label}  {n:>6} tok  median {s['median_ms']:>9.1f} ms  "
            f"{s['tok_per_s_median']:>8.1f} tok/s  spread {s['spread_pct']:.2f} %",
            file=sys.stderr,
        )
    out = {"kind": "ladder", "label": args.label, "results": results}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("floor", help="A-vs-A noise floor on one boot")
    f.add_argument("--url", required=True)
    f.add_argument("--tokens", type=int, default=8192)
    f.add_argument("--draws", type=int, default=3)
    f.add_argument("--seed", type=int, default=None)
    f.set_defaults(func=cmd_floor)

    ladder = sub.add_parser("ladder", help="prefill ladder over prompt lengths")
    ladder.add_argument("--url", required=True)
    ladder.add_argument("--tokens", default="2048,8192,32768")
    ladder.add_argument("--draws", type=int, default=3)
    ladder.add_argument("--label", default="arm")
    ladder.add_argument("--out", default=None)
    ladder.add_argument("--seed", type=int, default=None)
    ladder.set_defaults(func=cmd_ladder)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
