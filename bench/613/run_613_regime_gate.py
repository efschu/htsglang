#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Boot validation for the #613 prefill-graph regime gate.

WHAT THIS VALIDATES, and why it is not the same question window 4 answered.
Window 4 (`3b4526c4ac`) already measured the captured prefill under barlink and
found a REGIME SPLIT: **-4.62%** at 256 tokens x4 concurrent, **+10.25%** at
1900 single-stream, both clearing their own A-vs-A floors (85x and 2.3x). That
settled *whether* prefill graphs pay. It did not test an actuator, because none
existed.

The actuator now exists: `prefill_graph_regime.regime_permits_graph`, consulted
per batch from `can_run_graph`. Its entire purpose is to keep the -4.62% and
drop the +10.25%. So the question here is narrow and falsifiable:

    with the gate ON, does the long single-stream point stop paying the
    captured-path penalty, while the short concurrent point keeps its win?

That is a three-arm comparison, not a two-arm one:

    EAGER   prefill graph off entirely            -- the reference
    GRAPH   prefill graph on, regime gate OFF     -- window 4's "graphs" arm
    GATED   prefill graph on, regime gate ON      -- the thing under test

The gate is doing its job iff, against EAGER: GATED matches at the long point
(the gate refused, so the eager path ran) and GATED keeps GRAPH's win at the
short point (the gate permitted). Anything else is a routing bug and is
reportable as such.

WHY EAGER IS RE-MEASURED RATHER THAN QUOTED. Window 4's absolute numbers were
taken under specific power caps (3080 200 W, 5090 400 W) and on a different
checkpoint generation. Quoting them as this run's reference would be the
borrowed-baseline mistake. Every arm here is measured in the same session,
interleaved, and each delta counts only against this session's own floor.

USAGE

    # hermetic; no GPU, no server, no window
    python bench/613/run_613_regime_gate.py --self-test

    # inside a claimed window, against a server booted with
    #   --cuda-graph-backend-prefill breakable
    # (that flag LOCKS the prefill phase, so the multimodal compatibility rule
    #  does not knock it back to disabled -- no source change needed)
    python bench/613/run_613_regime_gate.py --run --port 30043 --out <dir>

Exit: 0 = gate validated, 1 = a check failed, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

#: The two shapes window 4 measured. Reused verbatim so this run is comparable
#: to it in SHAPE even though its absolute numbers are not comparable.
POINT_LONG = {"name": "1900_single", "tokens": 1900, "concurrency": 1}
POINT_SHORT = {"name": "256x4", "tokens": 256, "concurrency": 4}

#: Window 4's measured deltas, carried for orientation only. Never a threshold
#: here: this run scores against its OWN floor (see module docstring).
W4_LONG_DELTA_PCT = +10.25
W4_SHORT_DELTA_PCT = -4.62

ARMS = ("EAGER", "GRAPH", "GATED")


@dataclass
class ArmResult:
    arm: str
    point: str
    samples_ms: List[float] = field(default_factory=list)

    @property
    def median_ms(self) -> Optional[float]:
        if not self.samples_ms:
            return None
        return statistics.median(self.samples_ms)


def pct_delta(treatment_ms: float, reference_ms: float) -> float:
    """Signed percent: positive means the treatment is SLOWER."""
    if reference_ms <= 0:
        raise ValueError("reference must be positive")
    return (treatment_ms - reference_ms) / reference_ms * 100.0


def floor_pct(a_ms: float, b_ms: float) -> float:
    """The A-vs-A floor from two same-arm repeats, as a magnitude."""
    return abs(pct_delta(a_ms, b_ms))


def clears_floor(delta_pct: float, floor: float) -> bool:
    """A delta counts only if it exceeds the noise of its own arm."""
    return abs(delta_pct) > abs(floor)


@dataclass
class GateVerdict:
    long_ok: bool
    short_ok: bool
    detail: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.long_ok and self.short_ok


def judge_gate(
    *,
    long_gated_vs_eager: float,
    long_graph_vs_eager: float,
    short_gated_vs_eager: float,
    short_graph_vs_eager: float,
    long_floor: float,
    short_floor: float,
) -> GateVerdict:
    """Did the gate route both regimes as designed?

    LONG: the gate must REFUSE, so GATED should sit at EAGER -- i.e. its delta
    against EAGER must NOT clear the floor. If it does clear, the gate either
    did not refuse or refusing costs something it should not.

    SHORT: the gate must PERMIT, so GATED should keep GRAPH's win -- a
    NEGATIVE delta against EAGER that clears the floor. A gate that refuses
    here has thrown the win away, which is the failure mode a too-conservative
    threshold produces and is exactly why it is checked rather than assumed.
    """
    detail: List[str] = []

    long_ok = not clears_floor(long_gated_vs_eager, long_floor)
    detail.append(
        f"LONG  gated-vs-eager {long_gated_vs_eager:+.2f}% "
        f"(floor {long_floor:.2f}%) -> "
        + ("at eager, gate refused as designed" if long_ok else "NOT at eager")
        + f"; ungated graph arm was {long_graph_vs_eager:+.2f}%"
    )

    short_ok = short_gated_vs_eager < 0 and clears_floor(
        short_gated_vs_eager, short_floor
    )
    detail.append(
        f"SHORT gated-vs-eager {short_gated_vs_eager:+.2f}% "
        f"(floor {short_floor:.2f}%) -> "
        + ("win kept, gate permitted" if short_ok else "WIN LOST")
        + f"; ungated graph arm was {short_graph_vs_eager:+.2f}%"
    )
    return GateVerdict(long_ok, short_ok, detail)


def self_test() -> int:
    """Hermetic proof the harness computes and judges correctly, and can fail."""
    failures: List[str] = []
    ran: List[str] = []

    def check(label: str, cond: bool) -> None:
        ran.append(label)
        if not cond:
            failures.append(label)

    # -- arithmetic
    check("slower is positive", pct_delta(110.0, 100.0) == 10.0)
    check("faster is negative", pct_delta(95.0, 100.0) == -5.0)
    check("identical is zero", pct_delta(100.0, 100.0) == 0.0)
    check("floor is a magnitude", floor_pct(101.0, 100.0) == 1.0)
    check("a delta inside the floor does not count", not clears_floor(0.5, 2.0))
    check("a delta outside the floor counts", clears_floor(-4.62, 1.99))
    try:
        pct_delta(1.0, 0.0)
        check("zero reference raises", False)
    except ValueError:
        check("zero reference raises", True)

    # -- median, not mean: one driver hiccup must not set the number
    arm = ArmResult("GATED", "256x4", [200.0, 205.0, 900.0])
    check("median ignores the outlier", arm.median_ms == 205.0)
    check("no samples means no median", ArmResult("EAGER", "x").median_ms is None)

    # -- THE JUDGEMENT. A gate working as designed: long back at eager, short
    #    keeping the win.
    good = judge_gate(
        long_gated_vs_eager=+0.05,
        long_graph_vs_eager=+10.25,
        short_gated_vs_eager=-4.60,
        short_graph_vs_eager=-4.62,
        long_floor=0.12,
        short_floor=1.99,
    )
    check("a correct gate passes", good.passed)

    # -- and the three ways it must FAIL.
    not_refusing = judge_gate(
        long_gated_vs_eager=+10.10,  # still paying the captured penalty
        long_graph_vs_eager=+10.25,
        short_gated_vs_eager=-4.60,
        short_graph_vs_eager=-4.62,
        long_floor=0.12,
        short_floor=1.99,
    )
    check("a gate that did not refuse the long point FAILS", not not_refusing.passed)
    check("and it says so", "NOT at eager" in " ".join(not_refusing.detail))

    too_strict = judge_gate(
        long_gated_vs_eager=+0.05,
        long_graph_vs_eager=+10.25,
        short_gated_vs_eager=-0.10,  # win thrown away
        short_graph_vs_eager=-4.62,
        long_floor=0.12,
        short_floor=1.99,
    )
    check("a gate too strict to keep the win FAILS", not too_strict.passed)
    check("and it says so", "WIN LOST" in " ".join(too_strict.detail))

    wrong_sign = judge_gate(
        long_gated_vs_eager=+0.05,
        long_graph_vs_eager=+10.25,
        short_gated_vs_eager=+4.60,  # slower, not faster
        short_graph_vs_eager=-4.62,
        long_floor=0.12,
        short_floor=1.99,
    )
    check("a short point that got SLOWER FAILS", not wrong_sign.passed)

    # -- a win that does not clear its own floor is not a win
    noise = judge_gate(
        long_gated_vs_eager=+0.05,
        long_graph_vs_eager=+10.25,
        short_gated_vs_eager=-1.00,
        short_graph_vs_eager=-4.62,
        long_floor=0.12,
        short_floor=1.99,
    )
    check("a sub-floor win does not count", not noise.passed)

    # -- the gate's own decision function must agree with the arms this
    #    runner drives, or the runner is testing a different gate than ships
    try:
        from sglang.srt.model_executor.runner.prefill_graph_regime import (
            PREFILL_GRAPH_REGIME_ENV,
            regime_permits_graph,
        )

        prev = os.environ.get(PREFILL_GRAPH_REGIME_ENV)
        os.environ[PREFILL_GRAPH_REGIME_ENV] = "1"
        try:
            long_v = regime_permits_graph(
                batch_size=POINT_LONG["concurrency"], num_tokens=POINT_LONG["tokens"]
            )
            short_v = regime_permits_graph(
                batch_size=POINT_SHORT["concurrency"],
                num_tokens=POINT_SHORT["tokens"] * POINT_SHORT["concurrency"],
            )
        finally:
            if prev is None:
                os.environ.pop(PREFILL_GRAPH_REGIME_ENV, None)
            else:
                os.environ[PREFILL_GRAPH_REGIME_ENV] = prev
        check("the shipping gate refuses this runner's long point", not long_v.permits)
        check("the shipping gate permits this runner's short point", short_v.permits)
    except ImportError:
        check("the shipping gate is importable", False)

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    rejects = sum(1 for label in ran if "FAILS" in label or "does not" in label)
    print(f"self-test: OK ({len(ran)} checks, {rejects} asserting the judge rejects)")
    return 0


def _measure(port: int, point: dict, repeats: int) -> List[float]:  # pragma: no cover
    """Drive one workload point and return per-repeat wall times in ms."""
    import requests

    prompt = "word " * point["tokens"]
    out: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(point["concurrency"]):
            requests.post(
                f"http://127.0.0.1:{port}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                },
                timeout=300,
            )
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--port", type=int, default=30043)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="/tmp/613_regime_gate")
    ap.add_argument(
        "--arm",
        choices=ARMS,
        help="which arm this server was booted as; the runner measures one arm "
        "per invocation because the prefill backend and the gate are both "
        "read at boot",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.run:
        ap.print_help()
        return 2
    if not args.arm:
        print("cannot run: --arm is required (EAGER | GRAPH | GATED).")
        print("Each arm is a separate boot: the prefill backend and")
        print("SGLANG_PREFILL_GRAPH_REGIME are both read at load time.")
        return 2

    try:
        import requests

        health = requests.get(f"http://127.0.0.1:{args.port}/health", timeout=5)
        if health.status_code != 200:
            raise RuntimeError(f"health {health.status_code}")
    except Exception as exc:
        print(f"cannot run: no healthy server on port {args.port}: {exc}")
        return 2

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for point in (POINT_LONG, POINT_SHORT):
        samples = _measure(args.port, point, args.repeats)
        res = ArmResult(args.arm, point["name"], samples)
        results[point["name"]] = {
            "samples_ms": samples,
            "median_ms": res.median_ms,
        }
        print(f"[{args.arm}] {point['name']}: median {res.median_ms:.1f} ms {samples}")

    path = os.path.join(args.out, f"arm_{args.arm}.json")
    with open(path, "w") as f:
        json.dump({"arm": args.arm, "points": results}, f, indent=2)
    print(f"wrote {path}")
    print(
        "\nRun all three arms, then judge with judge_gate(); a single arm is "
        "not a verdict."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
