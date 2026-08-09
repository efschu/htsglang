#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631: does a flip COMMIT while requests are resident and speculating,
and is the answer of a request that crosses the cutover still correct?

WHY A COUNTING TASK. Every earlier probe on this strand asked the model
for prose and then argued about whether the output "looked right" -- and
the no-flip control drifted EARLIER than the flip run, which is what
proved the drift was the model rather than the cutover (HANDOFF v3 §4).
A count is different in kind: it is a determined sequence, every position
is checkable, and a corruption is located AT the token where it happens.
Since the flip is triggered mid-decode, a carry that loses or garbles
state shows up as a break at a known place instead of as a vibe.

WHAT A PASS REQUIRES, all four together:

  1. the flip COMMITS with requests resident -- the whole point; every
     armed window under load abandoned by design before the bootstrap;
  2. every crossing request's count is unbroken and complete;
  3. the server is healthy afterwards, with no fault, no SIGQUIT and no
     collective timeout in the log;
  4. the TP phase is really speculating afterwards (accept length
     present), because a "carry" that silently left speculation off would
     satisfy 1-3 and be worthless.

Reads the serving log for 1, 3 and 4 rather than trusting HTTP: a leg that
parks and abandons also returns 200 (the lesson that made a refused leg
read once as a green round trip).
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

PORT = 30030
LOG = "/spinning/serving-30030.boot.log"
COUNT_TO = 120


def post(path, payload, timeout=180):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def health():
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/health", timeout=5
        ) as r:
            return r.status
    except Exception:
        return 0


def flip(direction):
    try:
        return post("/phase_flip", {"direction": direction}, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def count_request(idx, results):
    """A long deterministic decode: the numbers 1..COUNT_TO in order."""
    prompt = (
        "Write the integers from 1 to %d in order, separated by single "
        "spaces, and write nothing else at all.\nAnswer: 1" % COUNT_TO
    )
    t0 = time.time()
    try:
        out = post(
            "/generate",
            {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 600,
                    "temperature": 0,
                },
            },
        )
        results[idx] = {
            "text": out.get("text", ""),
            "meta": out.get("meta_info", {}),
            "s": round(time.time() - t0, 1),
        }
    except Exception as exc:  # noqa: BLE001
        results[idx] = {"error": str(exc), "s": round(time.time() - t0, 1)}


def check_count(text):
    """Return (ok, detail). The prompt primes '1', so the reply starts at 2."""
    nums = [int(x) for x in re.findall(r"\d+", text or "")]
    if not nums:
        return False, "no digits in reply"
    want = list(range(nums[0], nums[0] + len(nums)))
    if nums == want and nums[0] <= 3 and nums[-1] >= COUNT_TO - 2:
        return True, f"{nums[0]}..{nums[-1]} unbroken ({len(nums)} numbers)"
    for i, (a, b) in enumerate(zip(nums, want)):
        if a != b:
            return False, (
                f"BREAK at position {i}: got {a}, expected {b}; "
                f"reply starts {nums[:6]} ends {nums[-6:]}"
            )
    return False, f"short: {nums[0]}..{nums[-1]} ({len(nums)} numbers)"


def log_slice(since_line):
    try:
        with open(LOG, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return [], since_line
    return lines[since_line:], len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--delay", type=float, default=4.0,
                    help="seconds of decode before the flip is armed")
    args = ap.parse_args()

    with open(LOG, "r", errors="replace") as fh:
        mark = len(fh.readlines())

    verdicts = []
    for cycle in range(args.cycles):
        direction = "pp_to_tp" if cycle % 2 == 0 else "tp_to_pp"
        print(f"\n=== cycle {cycle}: {direction} under load ===", flush=True)

        results = {}
        threads = [
            threading.Thread(target=count_request, args=(i, results))
            for i in range(args.concurrency)
        ]
        for t in threads:
            t.start()
        time.sleep(args.delay)
        print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] arming "
              f"{direction} with {args.concurrency} request(s) decoding",
              flush=True)
        resp = flip(direction)
        print(f"  /phase_flip -> {json.dumps(resp)[:200]}", flush=True)
        for t in threads:
            t.join()

        ok_all = True
        for i in sorted(results):
            r = results[i]
            if "error" in r:
                print(f"  req{i}: ERROR {r['error']}")
                ok_all = False
                continue
            ok, detail = check_count(r["text"])
            ok_all = ok_all and ok
            print(f"  req{i}: {'OK ' if ok else 'BAD'} {detail}  "
                  f"({r['s']}s, {r['meta'].get('completion_tokens')} tok)")
        verdicts.append((direction, ok_all))

    new, _ = log_slice(mark)
    joined = "".join(new)
    facts = {
        "bootstrapped": len(re.findall(r"PHASE-FLIP-DRAFT bootstrapped", joined)),
        "retuned": len(re.findall(r"retuned \d+ carried batch", joined)),
        "cutovers": len(re.findall(r"cutover", joined, re.I)),
        "abandoned": len(re.findall(r"FLIP ABANDONED", joined)),
        "faults": len(re.findall(r"illegal memory access|SIGQUIT|Fatal", joined)),
        "collective_timeouts": len(re.findall(r"no progress for", joined)),
        "accept_len": re.findall(r"accept_len[: =]+([0-9.]+)", joined)[-5:],
    }
    print("\n=== log facts since probe start ===")
    for k, v in facts.items():
        print(f"  {k}: {v}")
    print(f"  health: {health()}")

    ok = (
        all(v for _, v in verdicts)
        and facts["bootstrapped"] > 0
        and facts["faults"] == 0
        and facts["collective_timeouts"] == 0
        and health() == 200
    )
    print("\nVERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
