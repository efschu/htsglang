#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631: speculative accept length ON THE WIRE, and the post-long-session flip.

TWO THINGS AT ONCE, deliberately, because they need the same traffic.

1. THE WIRE GATE. The acceptance asks for accept length as a CLIENT-VISIBLE
   fact, not a scheduler log line. The OpenAI chat route does not carry it:
   ``/v1/chat/completions`` returns an empty ``meta_info``. The native
   ``/generate`` route does -- but only for a request that actually
   SPECULATED, and this instance speculates only in its TP phase. A single
   short request at rest runs entirely in PP, draws no drafts, and returns
   no counters: correct behaviour that looks exactly like a missing wire.
   So this issues CONCURRENT DECODE work, which is what moves the policy
   into TP, and reports the counters from whichever requests verified there.

2. THE CRASH REPRODUCTION. Concurrent ``/generate`` issued minutes after a
   very long session is the exact recipe that took all three ranks down in
   ``build_flip_live_slots_fn`` (a Req admitted but not yet allocated made
   ``req_to_token[None, :n]`` a 3-D tensor). Run directly after the needle
   probe, this is that reproduction; a clean pass with the instance still
   healthy afterwards is the fix's metal proof.

Exit status is the verdict: 0 only when the server stayed healthy AND at
least one response carried spec counters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

PROMPT = (
    "Explain, step by step and with concrete examples, how pipeline "
    "parallelism and tensor parallelism differ in an inference server, "
    "and when each one is the better choice."
)


def one(port: int, max_new: int, timeout: float, idx: int):
    t0 = time.time()
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/generate",
            json={
                "text": PROMPT,
                "sampling_params": {
                    "max_new_tokens": max_new,
                    "temperature": 0.0,
                },
            },
            timeout=timeout,
        )
        r.raise_for_status()
        mi = r.json().get("meta_info", {}) or {}
        spec = {k: v for k, v in mi.items() if k.startswith("spec_")}
        return {
            "idx": idx,
            "ok": True,
            "seconds": round(time.time() - t0, 2),
            "completion_tokens": mi.get("completion_tokens"),
            "spec": spec,
        }
    except Exception as exc:  # noqa: BLE001 - the failure IS the result
        return {
            "idx": idx,
            "ok": False,
            "seconds": round(time.time() - t0, 2),
            "error": f"{type(exc).__name__}: {exc}",
            "spec": {},
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    report = {"rounds": [], "health_before": None, "health_after": None}

    def health():
        try:
            return requests.get(
                f"http://127.0.0.1:{args.port}/health", timeout=5
            ).status_code
        except Exception:  # noqa: BLE001
            return 0

    report["health_before"] = health()

    for rnd in range(args.rounds):
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            res = list(
                pool.map(
                    lambda i: one(
                        args.port, args.max_new_tokens, args.timeout, i
                    ),
                    range(args.concurrency),
                )
            )
        report["rounds"].append(res)
        for r in res:
            print(
                json.dumps(
                    {
                        "round": rnd,
                        "idx": r["idx"],
                        "ok": r["ok"],
                        "seconds": r["seconds"],
                        "spec": r.get("spec"),
                        **({"error": r["error"]} if not r["ok"] else {}),
                    }
                ),
                flush=True,
            )

    report["health_after"] = health()

    with_spec = [
        r
        for rnd in report["rounds"]
        for r in rnd
        if r.get("spec", {}).get("spec_accept_length") is not None
    ]
    failures = [r for rnd in report["rounds"] for r in rnd if not r["ok"]]
    report["requests_with_wire_accept_len"] = len(with_spec)
    report["failed_requests"] = len(failures)
    if with_spec:
        report["wire_accept_length"] = with_spec[0]["spec"].get(
            "spec_accept_length"
        )
        report["wire_accept_rate"] = with_spec[0]["spec"].get("spec_accept_rate")

    out = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
    print(out)
    healthy = report["health_after"] == 200 and not failures
    print(
        f"VERDICT healthy_after={healthy} "
        f"wire_accept_len_requests={len(with_spec)} "
        f"accept_length={report.get('wire_accept_length')}"
    )
    return 0 if (healthy and with_spec) else 1


if __name__ == "__main__":
    sys.exit(main())
