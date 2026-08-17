#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#709 -- turnkey A/B for uneven-TP proportional sharding in TP decode.

    # boot arm A (current config), then:
    python bench/709/run_709_ab.py --arm A_equal --boot-id $(date +%s) \
        --clock-json /tmp/clockA.json --out /tmp/709_A.json
    # reboot with --rank-tp-ratio 2,1,1, then:
    python bench/709/run_709_ab.py --arm B_proportional --boot-id $(date +%s) \
        --clock-json /tmp/clockB.json --out /tmp/709_B.json
    python bench/709/run_709_ab.py --report /tmp/709_A.json /tmp/709_B.json

    python bench/709/run_709_ab.py --dry-run     # no server, no GPU

THIS SCRIPT DOES NOT BOOT ANYTHING. Boots belong to the serving lane; it
attaches to whatever is already running on --port, drives load, and measures.

WHAT IT MEASURES ITSELF: end-to-end decode round timing at bs 1-4, an A-vs-A
noise floor in THIS boot before any comparison, and a determined-answer
coherence digest (temperature 0, fixed prompts) -- because this lever is
lossless and a changed answer voids a speed win.

WHAT IT REFUSES TO INVENT: the per-rank COMPUTE/WAIT split. That lives in the
server's own CollectiveClock, not over HTTP, and it is the ONLY resolvable
discriminator here -- #705 predicts +0.780 ms on a ~30 ms round, i.e. 2.6%
against a 14.1% floor, so the end-to-end number this script can measure by
itself CANNOT answer the question. The window supplies the per-rank numbers via
--clock-json and the harness refuses to judge without them, rather than
returning a confident null from the metric that cannot see the effect.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acceptance as A  # noqa: E402

COHERENCE_PROMPTS = [
    "List the first five prime numbers, comma separated.",
    "What is 17 multiplied by 23? Answer with the number only.",
    "Name the capital of Japan. One word.",
]


def _post(port: int, path: str, payload: dict, timeout: float = 120.0):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode())


def coherence_digest(port: int, max_new: int = 24) -> str:
    """Determined answers, hashed. Temperature 0 and a fixed prompt set, so a
    digest change means the LEVER changed the answer -- which voids it."""
    parts = []
    for p in COHERENCE_PROMPTS:
        r = _post(port, "/generate", {
            "text": p,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
        })
        parts.append(r["text"] if isinstance(r, dict) else str(r))
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def timed_decode(port: int, batch: int, seconds: float, max_new: int = 64):
    """Round time under sustained decode load, bounded by TIME not by count."""
    payload = {
        "text": ["Count slowly from one to forty." for _ in range(batch)],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
    }
    _post(port, "/generate", payload)  # warm-up, not counted
    t0 = time.perf_counter()
    rounds = 0
    while time.perf_counter() - t0 < seconds:
        _post(port, "/generate", payload)
        rounds += 1
    elapsed = time.perf_counter() - t0
    return (elapsed / max(rounds, 1)) * 1000.0 / max(max_new, 1), elapsed


def measure_floor(port: int, batch: int, seconds: float) -> float:
    """A-vs-A IN THIS BOOT. Never imported from another boot: --rank-tp-ratio
    is a boot flag, so the arms are already cross-boot and the floor is the
    only thing that can still be same-boot."""
    a, _ = timed_decode(port, batch, seconds)
    b, _ = timed_decode(port, batch, seconds)
    return abs(a - b) / max(min(a, b), 1e-9)


def load_clock(path: str, arm: str, batch_rounds):
    """Per-rank COMPUTE/WAIT, supplied by the window from the server side.

    Schema (one object per rank per batch):
      [{"rank":0,"card":"5090","batch":1,"ms_round":30.1,
        "ms_compute":27.0,"ms_wait":3.1,"seconds_measured":12.0}, ...]
    """
    with open(path) as fh:
        rows = json.load(fh)
    if not rows:
        raise SystemExit(f"{path} is empty; the clock split is the discriminator")
    return [A.RankRound(arm=arm, **r) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A_equal", "B_proportional"])
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--boot-id")
    ap.add_argument("--ratio-flag", default=None,
                    help="verbatim --rank-tp-ratio of the RUNNING server")
    ap.add_argument("--clock-json", help="per-rank COMPUTE/WAIT from the server")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="709_arm.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", nargs="*")
    args = ap.parse_args()

    if args.report:
        arms = {}
        for path in args.report:
            with open(path) as fh:
                d = json.load(fh)
            arms[d["arm"]] = A.ArmRun(
                arm=d["arm"], ratio_flag=d["ratio_flag"], boot_id=d["boot_id"],
                own_noise_floor=d["own_noise_floor"],
                rounds=[A.RankRound(**r) for r in d["rounds"]],
                answer_digest=d["answer_digest"],
            )
        if len(arms) != 2:
            raise SystemExit(f"need both arms, got {sorted(arms)}")
        print(A.render(A.evaluate(arms["A_equal"], arms["B_proportional"])))
        return 0

    if args.dry_run:
        # MOCK-SMOKE: no server, no GPU. Exercises the schema, the acceptance
        # path, and the refusals -- including that the unresolvable metric
        # cannot on its own produce a CONFIRM.
        a = A.ArmRun("A_equal", "None", "boot1", 0.05,
                     [A.RankRound("A_equal", i, f"c{i}", 1, 30.0, 30.0 - w, w, 12.0)
                      for i, w in enumerate((0.0, 3.0, 3.0))], "digest0")
        b = A.ArmRun("B_proportional", "2,1,1", "boot2", 0.05,
                     [A.RankRound("B_proportional", i, f"c{i}", 1, 29.2,
                                  29.2 - w, w, 12.0)
                      for i, w in enumerate((0.2, 0.4, 0.3))], "digest0")
        rep = A.evaluate(a, b)
        assert rep["confirm"], "dry-run: a collapsing wait spread must CONFIRM"
        assert not rep["secondary_not_the_discriminator"]["end_to_end_is_resolvable"]
        bad = dataclasses.replace(b, answer_digest="CHANGED")
        assert not A.evaluate(a, bad)["confirm"], "dry-run: coherence must gate"
        print(A.render(rep))
        print("  dry-run: acceptance path, coherence gate and the "
              "unresolvable-metric flag all exercised; no server touched")
        return 0

    for required in ("arm", "boot_id", "clock_json", "ratio_flag"):
        if getattr(args, required) in (None, ""):
            raise SystemExit(
                f"--{required.replace('_', '-')} is required. In particular the "
                "ratio flag must be the RUNNING server's verbatim value: 'auto' "
                "is CAPACITY-first (VRAM ratio ~1.6:1:1), not the "
                "bandwidth-proportional ~2.36:1:1 that #705's +0.780 ms was "
                "derived from, and an unrecorded arm cannot be judged."
            )

    print(f"arm {args.arm}  port {args.port}  ratio {args.ratio_flag}")
    floor = measure_floor(args.port, 1, min(args.seconds, 5.0))
    print(f"A-vs-A floor measured IN THIS BOOT: {floor:.4f}")
    digest = coherence_digest(args.port)
    print(f"coherence digest: {digest}")
    for batch in (1, 2, 3, 4):
        ms, secs = timed_decode(args.port, batch, args.seconds)
        print(f"  bs={batch}  {ms:8.4f} ms/token-round over {secs:5.1f}s")
    rounds = load_clock(args.clock_json, args.arm, None)
    payload = {
        "arm": args.arm, "ratio_flag": args.ratio_flag, "boot_id": args.boot_id,
        "own_noise_floor": floor, "answer_digest": digest,
        "rounds": [dataclasses.asdict(r) for r in rounds],
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}; run --report with both arms to get the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
