#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#658: drive the #330 budget dial DOWN and back UP on a live instance.

WHAT THIS PROVES, AND WHAT IT CANNOT
------------------------------------
The desk half of #658 is hermetic: the corridor law is a term in the dial's
floor, and a reduction spends the guard's relief ladder before the capacity
arithmetic. Neither fact is worth anything until an external tenant can lower
this instance's budget on metal, watch the driver's free column move, raise it
again, and see serving carry on -- with the corridor law unbroken at every
100 ms sample throughout.

So this script is deliberately an EXTERNAL TENANT and nothing else. It talks
to the same public endpoint a video lane or a training tenant would use
(``POST /vram_budget``), reads NVML from outside the process, and never
imports a single sglang symbol. A probe that reached inside would prove the
mechanism to itself.

THE CORRIDOR SAMPLER IS THE JUDGE, NOT THIS SCRIPT. The law is a CONTINUOUS
minimum and this script samples at seconds; ``corridor_sample.sh`` runs at
100 ms alongside and owns the breach verdict. What is recorded here is the
STEP: the free column before the cut, after the cut, and after the restore,
plus the dial's own status at each point, so the NVML movement can be
attributed to the dial rather than to whatever the load happened to do.

WHY THE RAISE IS PART OF THE SAME RUN. A mechanism that only gives memory
away is a leak with a nice interface. The restore leg is what makes the dial
a DIAL, and spec item 13's requirement -- that restored sessions come back on
CUDA graphs -- is only observable after a raise. In this lane the graphs are
never left at all (the VA is stable and the store bound is baked to cover the
grown ceiling at capture time), so what this leg checks is the OBSERVABLE
consequence: the instance still answers, at the same capacity it started
with, with no re-capture in between.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request

BASE_DEFAULT = "http://127.0.0.1:30030"


def _post(base: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        f"{base}/vram_budget",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _nvml_free():
    """Per-card FREE in MiB, from outside the process.

    The FREE column, never total-minus-used: the driver holds a per-card
    carve-out back from both, so the subtraction over-states free by exactly
    that amount and the corridor law is written on the column.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [int(x.strip()) for x in out.split("\n") if x.strip()]


def _health(base: str) -> int:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=15) as r:
            return r.status
    except Exception:
        return 0


def _step(base: str, label: str, rec: list):
    free = _nvml_free()
    try:
        st = _post(base, {"query": True}, timeout=60)
    except Exception as e:
        st = {"error": str(e)}
    row = {
        "t": time.strftime("%H:%M:%SZ", time.gmtime()),
        "label": label,
        "nvml_free_mib": free,
        "health": _health(base),
        "status": st,
    }
    rec.append(row)
    print(f"[{row['t']}] {label}: free={free} MiB health={row['health']}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=BASE_DEFAULT)
    ap.add_argument(
        "--release-mib",
        type=int,
        default=1024,
        help="how much an external tenant claims, per rank",
    )
    ap.add_argument(
        "--settle",
        type=float,
        default=90.0,
        help="seconds to wait after each dial for the consensus boundary; the "
        "capacity commit needs a group-wide IDLE boundary, so under load "
        "this must be generous or the residual simply has not committed yet",
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rec: list = []
    print(f"===== #658 BUDGET CYCLE against {a.url}, {a.release_mib} MiB/rank")
    if _health(a.url) != 200:
        print("REFUSE: the instance is not healthy before the cycle", file=sys.stderr)
        return 2

    _step(a.url, "baseline", rec)

    # -- DOWN: an external tenant claims memory --------------------------------
    print(f"-- dialing DOWN by {a.release_mib} MiB per rank")
    t0 = time.time()
    down = _post(a.url, {"device": "all", "release_mib": a.release_mib})
    print(f"   reply: {json.dumps(down)[:400]}")
    rec.append({"label": "dial_down_reply", "reply": down, "ms": (time.time() - t0) * 1000})
    time.sleep(a.settle)
    _step(a.url, "after_down", rec)

    # -- UP: the tenant gives it back ------------------------------------------
    print(f"-- dialing UP by {a.release_mib} MiB per rank")
    t0 = time.time()
    up = _post(a.url, {"device": "all", "release_mib": -a.release_mib})
    print(f"   reply: {json.dumps(up)[:400]}")
    rec.append({"label": "dial_up_reply", "reply": up, "ms": (time.time() - t0) * 1000})
    time.sleep(a.settle)
    _step(a.url, "after_up", rec)

    ok = _health(a.url) == 200
    print(f"===== CYCLE {'OK' if ok else 'FAILED'}: health {_health(a.url)}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"written: {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
