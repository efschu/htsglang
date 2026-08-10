#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#656 spec item 8: the draft-CUDA-graph A/B, measured on the wire.

WHY THIS CANNOT BE A SAME-BOOT A/B, and what actually controls the noise.
``--disable-draft-cuda-graph`` is a boot-time server argument: the graphs
are captured once during worker init, so the two arms are two boots and
the usual same-boot floor is unavailable. The bench-variance law of this
chain says decode throughput tracks output CONTENT with r=0.90, so the
first design here hashed the response text and planned to void any
comparison whose text diverged.

MEASURED, and it killed that plan: at temperature 0.0, with the same
prompt, THIS INSTANCE DOES NOT REPRODUCE ITS OWN OUTPUT WITHIN A SINGLE
BOOT. Round 1 and round 2 of the graphs-ON arm returned different text
for the identical request (2565 vs 2741 characters, idx 3). Batch
composition varies with arrival timing, and a speculative decode's
verification path is not batch-invariant. So text equality is not
achievable even A-vs-A, and a control nothing can pass is not a control.

What replaces it, and it is weaker on purpose because the honest control
is weaker:
  1. TOKEN COUNT IS PINNED. Every request runs to ``max_new_tokens``, so
     each arm decodes exactly the same number of tokens. Content still
     varies, but not the amount of work counted.
  2. THE A-vs-A FLOOR IS MANDATORY. Run this arm twice against the SAME
     boot before comparing across boots. A cross-boot delta smaller than
     the same-boot spread is noise, and ``--compare`` prints the floor
     alongside the delta so the two are read together and never apart.
The hashes are still recorded: they cannot validate an arm, but they
document that the divergence is real rather than assumed.

WHAT IS MEASURED, and why accept length is computed and not read.
``meta_info.spec_accept_length`` is a per-request figure. Averaging those
averages weights a 40-token request like a 600-token one. The aggregate
reported here is

    accept_len = sum(completion_tokens) / sum(spec_verify_ct)

which is the definition, over the whole arm. The per-request values are
kept in the JSON so a wide spread is visible rather than hidden by its
own mean.

WHY THE TRAFFIC IS CONCURRENT: this instance speculates only in its TP
phase, and under strict purity a lone short request runs entirely in PP,
draws no drafts and returns no counters -- a correct instance that reads
exactly like a broken wire. Concurrency is what moves the policy into TP.

Usage:
  s28_graph_ab_probe.py --out /spinning/evidence-631/s28/arm_on.json
  s28_graph_ab_probe.py --compare arm_on.json arm_off.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# Fixed and varied: one prompt per concurrent slot, so the slots do not all
# decode the same token stream (which would make the batch unrepresentative
# of real mixed traffic), but slot i asks the same question in both arms.
PROMPTS = [
    "Explain, step by step and with concrete examples, how pipeline "
    "parallelism and tensor parallelism differ in an inference server, "
    "and when each one is the better choice.",
    "Describe in detail how a key-value cache works in transformer "
    "inference, why it grows with context length, and what strategies "
    "exist to bound its memory footprint.",
    "Write a thorough explanation of speculative decoding: the draft "
    "model, the verification step, why it preserves the target "
    "distribution, and where the speedup actually comes from.",
    "Give a detailed account of how CUDA graphs reduce kernel launch "
    "overhead, what constraints they impose on the captured region, and "
    "when capturing a region is not worth it.",
]


def one(port: int, prompt: str, max_new: int, timeout: float, idx: int):
    t0 = time.time()
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new,
                    "temperature": 0.0,
                },
            },
            timeout=timeout,
        )
        r.raise_for_status()
        body = r.json()
        mi = body.get("meta_info", {}) or {}
        text = body.get("text", "")
        return {
            "idx": idx,
            "ok": True,
            "seconds": round(time.time() - t0, 3),
            "completion_tokens": mi.get("completion_tokens"),
            "prompt_tokens": mi.get("prompt_tokens"),
            "spec_verify_ct": mi.get("spec_verify_ct"),
            "spec_accept_length": mi.get("spec_accept_length"),
            "spec_accept_rate": mi.get("spec_accept_rate"),
            # The content control. Truncated for readability; 16 hex chars
            # is far past any collision risk over a few dozen responses.
            "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
            "text_len": len(text),
        }
    except Exception as exc:  # noqa: BLE001 - the failure IS the result
        return {
            "idx": idx,
            "ok": False,
            "seconds": round(time.time() - t0, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_arm(args) -> dict:
    def health():
        try:
            return requests.get(
                f"http://127.0.0.1:{args.port}/health", timeout=10
            ).status_code
        except Exception:  # noqa: BLE001
            return 0

    report = {
        "label": args.label,
        "concurrency": args.concurrency,
        "rounds": args.rounds,
        "max_new_tokens": args.max_new_tokens,
        "health_before": health(),
        "requests": [],
    }

    # Warmup round, discarded. The first concurrent batch after an idle
    # instance pays the policy's move into TP and, in the graphs-ON arm,
    # any lazily-touched capture; neither is a steady-state cost.
    if args.warmup:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(
                pool.map(
                    lambda i: one(
                        args.port, PROMPTS[i % len(PROMPTS)],
                        min(200, args.max_new_tokens), args.timeout, i,
                    ),
                    range(args.concurrency),
                )
            )
        report["warmup"] = "one discarded round"

    t_start = time.time()
    for rnd in range(args.rounds):
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            res = list(
                pool.map(
                    lambda i: one(
                        args.port, PROMPTS[i % len(PROMPTS)],
                        args.max_new_tokens, args.timeout, i,
                    ),
                    range(args.concurrency),
                )
            )
        for r in res:
            r["round"] = rnd
            report["requests"].append(r)
            print(json.dumps(r), flush=True)
    t_end = time.time()

    report["health_after"] = health()
    report["t_start"] = t_start
    report["t_end"] = t_end
    report["wall_seconds"] = round(t_end - t_start, 3)

    ok = [r for r in report["requests"] if r.get("ok")]
    report["failed"] = len(report["requests"]) - len(ok)
    tok = sum(r.get("completion_tokens") or 0 for r in ok)
    verify = sum(r.get("spec_verify_ct") or 0 for r in ok)
    report["total_completion_tokens"] = tok
    report["total_spec_verify_ct"] = verify
    # The definition, over the arm -- not a mean of per-request means.
    report["accept_len_aggregate"] = round(tok / verify, 4) if verify else None
    per = [
        r["spec_accept_length"]
        for r in ok
        if r.get("spec_accept_length") is not None
    ]
    if per:
        report["accept_len_per_request_mean"] = round(statistics.mean(per), 4)
        report["accept_len_per_request_min"] = round(min(per), 4)
        report["accept_len_per_request_max"] = round(max(per), 4)
    report["wall_decode_tok_s"] = (
        round(tok / (t_end - t_start), 2) if t_end > t_start else None
    )
    lat = [r["seconds"] for r in ok]
    if lat:
        report["req_seconds_mean"] = round(statistics.mean(lat), 3)
        report["req_seconds_median"] = round(statistics.median(lat), 3)
        report["req_seconds_max"] = round(max(lat), 3)
    report["content_hashes"] = {
        f"{r['round']}:{r['idx']}": r.get("text_sha") for r in ok
    }
    return report


def compare(path_a: str, path_b: str, floor_path: str | None) -> int:
    """Cross-boot delta, read against the same-boot floor and never alone.

    Without ``floor_path`` this prints a delta it cannot qualify, and says
    so. That is allowed but it is not a verdict: the whole lesson of the
    content control above is that this instance varies run to run, and a
    number with no floor under it cannot distinguish a mechanism from that
    variation.
    """
    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)
    floor = None
    if floor_path:
        with open(floor_path) as fh:
            floor = json.load(fh)

    ha, hb = a.get("content_hashes", {}), b.get("content_hashes", {})
    shared = sorted(set(ha) & set(hb))
    mismatched = [k for k in shared if ha[k] != hb[k]]
    print(f"=== {a.get('label')} (A) vs {b.get('label')} (B) ===")
    print(f"content: {len(shared)} shared slots, {len(mismatched)} with "
          f"different text (expected; see header -- not a validity check)")
    print(f"work pinned: A={a.get('total_completion_tokens')} tokens, "
          f"B={b.get('total_completion_tokens')} tokens")
    if floor is not None:
        print(f"same-boot floor from: {floor.get('label')}")
    else:
        print("NO FLOOR SUPPLIED -- deltas below are unqualified, not verdicts")

    verdicts = []

    def row(name, key, better_high=True):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            print(f"  {name:26s} {va} -> {vb}")
            return
        d = vb - va
        pct = (d / va * 100) if va else float("nan")
        fl = None
        if floor is not None and floor.get(key) is not None and va:
            fl = abs(floor[key] - va) / va * 100
        if fl is None:
            tag = "unqualified"
        elif abs(pct) <= fl:
            tag = f"WITHIN FLOOR (+-{fl:.1f}%) -> no measured effect"
        else:
            direction = "better" if ((d > 0) == better_high) else "worse"
            tag = f"exceeds floor (+-{fl:.1f}%) -> {direction}"
        verdicts.append((name, pct, fl, tag))
        print(f"  {name:26s} {va:10.3f} -> {vb:10.3f}  "
              f"({d:+.3f}, {pct:+.1f}%)  {tag}")

    print("metrics (A -> B):")
    row("accept_len_aggregate", "accept_len_aggregate")
    row("wall_decode_tok_s", "wall_decode_tok_s")
    row("req_seconds_mean", "req_seconds_mean", better_high=False)
    row("req_seconds_median", "req_seconds_median", better_high=False)
    row("req_seconds_max", "req_seconds_max", better_high=False)
    print(f"  failed A={a.get('failed')} B={b.get('failed')}")

    if floor is not None:
        real = [v for v in verdicts if v[2] is not None and abs(v[1]) > v[2]]
        print(f"\nSUMMARY: {len(real)} of {len(verdicts)} metrics move beyond "
              f"the same-boot floor.")
        if not real:
            print("  B buys nothing measurable over A on this rig.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--warmup", action="store_true", default=True)
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    ap.add_argument(
        "--floor", default=None,
        help="a REPEAT of arm A on the same boot; supplies the noise floor "
             "every cross-boot delta must clear to count as an effect",
    )
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare, args.floor)

    report = run_arm(args)
    out = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
    print(f"ARM {report['label']} accept_len={report['accept_len_aggregate']} "
          f"wall_tok_s={report['wall_decode_tok_s']} "
          f"req_s_mean={report.get('req_seconds_mean')} "
          f"failed={report['failed']} health={report['health_after']}")
    return 0 if report["failed"] == 0 and report["health_after"] == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
