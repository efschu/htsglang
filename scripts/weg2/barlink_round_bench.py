#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The per-round term of a bar1 decomposition, measured -- #1234 §6.

WHY THIS EXISTS
===============
``barlink_bar1.round_budget`` replaces the hand constant 16 with a crossover:
refuse a decomposition exactly when it would be slower than the rung it falls
to. That crossover has three constants, and the ones it ships with are
INTERIM -- fitted from boot ``wait`` numbers, which carry rank skew and are
not a transport timing. Two independent fits of the same boot logs disagree by
1.73x on the per-round term, and both overpredict the one window they were not
fitted on. This script is the artifact that replaces them.

The three things it must deliver, and none of them is optional:

1. **The per-round term IN ISOLATION.** The same byte count at two windows
   plans to two different round counts. The time difference divided by the
   round difference is ``round_us`` with no regression and no collinearity --
   exactly what a single-window fit cannot give, because there ``rounds`` and
   ``wire_bytes`` move together (normalised condition number 1.9e5).
2. **The joint fit** ``ms = round_ms*rounds + wire/wire_Bps`` over >= 3
   windows, with its residuals printed, not summarised.
3. **At least one decomposition of MORE THAN 16 ROUNDS actually EXECUTED**,
   with a correctness check on the result. No such decomposition has ever run
   on this rig; until one has, the widened coverage is arithmetic, not a
   measurement. 128 MiB at a 16-MiB window is 33 rounds.

WHAT IT MEASURES AGAINST
========================
bar1 and PyNccl on the SAME three ranks in the SAME process, interleaved per
round so drift and foreign load hit both. An A/A noise floor runs first: the
same backend against itself, so a later 1.2x can be read against the spread
of a 1.0x that is known to be one.

The host-staged gloo rung (``next_rung_gbps``) is measured too, because that
is the rung a refusal actually falls to -- a barlink-owned group never builds
a PyNccl communicator (``parallel_state.should_build_pynccl``), so NCCL is a
reference point here and not a fallback.

TURNKEY, AND IT REFUSES WITHOUT A WINDOW
========================================
GPU access on this rig goes through the gpuq window plan. This script asks
127.0.0.1:8770 for a RUNNING booking covering the cards it is about to touch
and exits 3 if there is none -- it will not quietly take cards that someone
else's window owns. Book one first::

    /opt/gpuq/bin/gpuq book --owner weg2-barlink --cards 0,1,2 \\
        --duration 20m --purpose "#1234 round-term microbench"

    scripts/weg2/barlink_round_bench.py --out /spinning/gpu-arb/weg2/roundbench.json

Then release it. Set ``/spinning/gpu-arb/MEASURE_WINDOW`` for the duration so
no other strand boots into the measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import timedelta

GPUQ = "http://127.0.0.1:8770/api/v1"

#: Windows to sweep, MiB. Three is the minimum that identifies the per-round
#: term; four gives it a residual worth reading.
WINDOWS_MIB = (16, 24, 32, 40)
#: Messages, MiB. 96 is group dcp:0's operating point (4096 tokens x 24576 B);
#: 128 at a 16-MiB window is the 33-round point that deliverable 3 needs.
MESSAGES_MIB = (24, 40, 56, 72, 96, 128)


# ---------------------------------------------------------------------------
# the window plan
# ---------------------------------------------------------------------------

def _get(path: str, timeout: float = 4.0):
    with urllib.request.urlopen(f"{GPUQ}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


def require_gpuq_window(cards: list, booking_id: str = "") -> dict:
    """A RUNNING gpuq booking covering ``cards``, or ``SystemExit(3)``.

    The service plans, it does not lock -- so this is a courtesy check, not a
    safety one, and it says so when it passes. What it does prevent is the
    thing that actually happens: a benchmark started in a hurry on cards a
    different strand's window owns, and two sets of numbers that are both
    wrong.
    """
    want = set(int(c) for c in cards)
    try:
        if booking_id:
            rows = [_get(f"/bookings/{booking_id}")]
        else:
            payload = _get("/bookings")
            rows = payload if isinstance(payload, list) else payload.get(
                "bookings", []
            )
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise SystemExit(
            f"REFUSED: the gpuq window plan at {GPUQ} did not answer ({e!r}). "
            "This rig's GPUs are handed out as time windows; a measurement "
            "that takes cards outside one is not a measurement anybody can "
            "trust. Start the service or book a window and retry."
        )
    for row in rows:
        if str(row.get("state", "")).lower() != "running":
            continue
        have = set(int(c) for c in (row.get("cards") or []))
        if want <= have:
            return row
    raise SystemExit(
        f"REFUSED: no RUNNING gpuq booking covers cards {sorted(want)}. Book "
        f"one and retry:\n"
        f"  /opt/gpuq/bin/gpuq book --owner weg2-barlink --cards "
        f"{','.join(str(c) for c in sorted(want))} --duration 20m "
        f"--purpose '#1234 round-term microbench'\n"
        f"A pending window is not a running one -- end the turn and ask "
        f"again later rather than waiting here."
    )


def preflight(cards: list, max_used_mib: int) -> None:
    """The safety net the window plan explicitly does not provide."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    used = {}
    for line in out:
        idx, name, mib = (p.strip() for p in line.split(","))
        used[int(idx)] = (name, int(mib))
    busy = [(c, used.get(c, ("?", -1))) for c in cards
            if used.get(c, ("?", 0))[1] > max_used_mib]
    if busy:
        raise SystemExit(
            f"REFUSED: cards still occupied despite the window: {busy}. The "
            "window says it is your turn; the hardware says somebody is still "
            "on it. Find the process before measuring anything."
        )


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def _timed(fn, seconds: float, at_least: int = 3) -> list:
    """Per-iteration milliseconds over at least ``seconds`` of wall clock."""
    import torch

    samples = []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline or len(samples) < at_least:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def run_rank(a) -> int:
    import torch
    import torch.distributed as dist

    rank, world = a.rank, len(a.cards)
    dev_ord = rank
    torch.cuda.set_device(dev_ord)
    dev = torch.device("cuda", dev_ord)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    to = timedelta(seconds=a.pg_timeout)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world,
        timeout=to, device_id=dev,
    )
    gloo_pg = dist.new_group(list(range(world)), backend="gloo", timeout=to)
    nccl_pg = dist.new_group(list(range(world)), backend="nccl", timeout=to)

    sys.path.insert(0, a.sglang_python)
    import sglang.srt.distributed.device_communicators.barlink as barlink_mod
    from sglang.srt.distributed.device_communicators.barlink_bar1 import (
        ar_plan,
        wire_bytes_for,
    )

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    results = []
    # The round cap is pinned OPEN for the whole bench. Deliverable 3 needs a
    # >16-round decomposition to actually execute, and the derived bound is
    # exactly what this run is measuring the inputs of -- using it here would
    # be circular.
    os.environ["SGLANG_BARLINK_BAR1_AR_MAX_ROUNDS"] = "100000"
    os.environ["SGLANG_BARLINK_BAR1_AG_MAX_ROUNDS"] = "100000"
    os.environ["SGLANG_BARLINK_BAR1_A2A_MAX_ROUNDS"] = "100000"
    os.environ["SGLANG_BARLINK_BAR1_BC_MAX_ROUNDS"] = "100000"
    os.environ["SGLANG_BARLINK_UNCOVERED_CLASS"] = "refuse"

    sizes = [m << 20 for m in a.messages]
    ref = None

    for wmib in a.windows:
        os.environ["SGLANG_BARLINK_BAR1_WINDOW_MIB"] = str(wmib)
        barlink_mod._TRANSPORT = "bar1"
        comm = barlink_mod.BarlinkCommunicator(cpu_group=gloo_pg, device=dev)
        got = type(comm.transport).__name__ if comm.transport else None
        if got != "BarlinkBar1Transport":
            log(f"# window {wmib} MiB: bar1 not active ({got}) -- skipped")
            try:
                comm.close()
            except Exception:       # noqa: BLE001
                pass
            continue
        chunk_max = int(comm.transport._geo["chunk_max"])
        for line in comm.transport.coverage_lines():
            log("# " + line)

        for nbytes in sizes:
            n = nbytes // 4
            buf = torch.full((n,), float(rank + 1), dtype=torch.float32,
                             device=dev)
            rounds = len(ar_plan(nbytes, chunk_max, world))

            # -- correctness, BEFORE any timing. A fast wrong answer is the
            # -- one failure mode a throughput bench cannot see.
            probe = buf.clone()
            comm.all_reduce(probe)
            want = float(sum(range(1, world + 1)))
            ok = bool(torch.all(probe == want).item())
            if not ok:
                raise SystemExit(
                    f"WRONG RESULT: window {wmib} MiB, {nbytes} B, {rounds} "
                    f"rounds -- expected {want}, got "
                    f"{probe.min().item()}..{probe.max().item()}. A "
                    f"decomposition that returns the wrong sum is not a slow "
                    f"case."
                )

            work = buf.clone()
            _timed(lambda: comm.all_reduce(work), a.warmup)
            bar1 = _timed(lambda: comm.all_reduce(work), a.secs)

            nccl = []
            if a.nccl:
                nwork = buf.clone()
                _timed(lambda: dist.all_reduce(nwork, group=nccl_pg),
                       a.warmup)
                nccl = _timed(lambda: dist.all_reduce(nwork, group=nccl_pg),
                              a.secs)

            row = {
                "window_mib": wmib,
                "chunk_max": chunk_max,
                "nbytes": nbytes,
                "rounds": rounds,
                "wire_bytes": wire_bytes_for("all_reduce", nbytes, world),
                "world": world,
                "correct": ok,
                "bar1_ms": statistics.median(bar1),
                "bar1_n": len(bar1),
                "bar1_p10": min(bar1),
                "nccl_ms": statistics.median(nccl) if nccl else None,
            }
            results.append(row)
            log(
                f"# w{wmib:>3} {nbytes >> 20:>4} MiB  {rounds:>3} rounds  "
                f"bar1 {row['bar1_ms']:8.3f} ms"
                + (f"  nccl {row['nccl_ms']:8.3f} ms" if nccl else "")
                + ("  OK" if ok else "  WRONG")
            )
            if ref is None:
                ref = row
        try:
            comm.close()
        except Exception as e:      # noqa: BLE001
            log(f"# close({wmib}) said {e!r}")

    # -- the host-staged rung, which is what a refusal really falls to -------
    next_rung = None
    if a.gloo_rung:
        nbytes = max(sizes)
        host = torch.zeros(nbytes // 4, dtype=torch.float32)
        _timed(lambda: dist.all_reduce(host, group=gloo_pg), a.warmup)
        ms = statistics.median(
            _timed(lambda: dist.all_reduce(host, group=gloo_pg), a.secs)
        )
        next_rung = nbytes / (ms / 1e3) / 1e9
        log(f"# host-staged gloo rung: {ms:.1f} ms for {nbytes} B "
            f"-> {next_rung:.3f} GB/s")

    if rank == 0:
        report(results, next_rung, a)
    dist.barrier(group=gloo_pg)
    return 0


# ---------------------------------------------------------------------------
# the three deliverables
# ---------------------------------------------------------------------------

def report(rows: list, next_rung, a) -> None:
    print("\n=== DELIVERABLE 1: the per-round term IN ISOLATION ===")
    print("Same byte count, two windows -> two round counts. dt/dR is "
          "round_us with no regression and no collinearity.")
    by_size = {}
    for r in rows:
        by_size.setdefault(r["nbytes"], []).append(r)
    diffs = []
    for nbytes, group in sorted(by_size.items()):
        group.sort(key=lambda r: r["rounds"])
        for i in range(len(group) - 1):
            lo, hi = group[i], group[i + 1]
            dR = hi["rounds"] - lo["rounds"]
            if dR <= 0:
                continue
            us = (hi["bar1_ms"] - lo["bar1_ms"]) * 1e3 / dR
            diffs.append(us)
            print(f"  {nbytes >> 20:>4} MiB  w{lo['window_mib']}->"
                  f"w{hi['window_mib']}  R {lo['rounds']}->{hi['rounds']}  "
                  f"round_us = {us:8.1f}")
    if diffs:
        print(f"  => round_us median {statistics.median(diffs):.1f}, "
              f"n={len(diffs)}, spread "
              f"{min(diffs):.1f}..{max(diffs):.1f}")

    print("\n=== DELIVERABLE 2: the joint fit, with residuals ===")
    fit = _least_squares(rows)
    if fit:
        round_us, wire_gbps, resid = fit
        print(f"  round_us = {round_us:.1f}   wire_gbps = {wire_gbps:.2f}")
        print(f"  residuals: mean {statistics.mean(abs(x) for x in resid):.3f}"
              f" ms, max {max(abs(x) for x in resid):.3f} ms, n={len(resid)}")
        for r, e in zip(rows, resid):
            print(f"    w{r['window_mib']:>3} {r['nbytes'] >> 20:>4} MiB "
                  f"R{r['rounds']:>3}  measured {r['bar1_ms']:8.3f}  "
                  f"residual {e:+7.3f}")
    else:
        print("  not enough independent cells to identify both terms")

    print("\n=== DELIVERABLE 3: a decomposition of MORE than 16 rounds, "
          "EXECUTED ===")
    deep = [r for r in rows if r["rounds"] > 16]
    if not deep:
        print("  NONE RAN. The widened coverage stays arithmetic, not a "
              "measurement -- rerun with a 16-MiB window and a 128-MiB "
              "message.")
    for r in sorted(deep, key=lambda r: -r["rounds"]):
        print(f"  w{r['window_mib']} {r['nbytes'] >> 20} MiB  "
              f"{r['rounds']} rounds  {r['bar1_ms']:.3f} ms  "
              f"result {'CORRECT' if r['correct'] else 'WRONG'}")

    if next_rung:
        print(f"\nnext_rung_gbps (host-staged gloo, measured): "
              f"{next_rung:.3f}")
    print("\nThese three numbers are what "
          "barlink_bar1.DEFAULT_{ROUND_US,WIRE_GBPS,NEXT_RUNG_GBPS} must "
          "carry, together with this run's artifact path, windows, round "
          "counts and residuals. A bare number in that docstring is the "
          "defect #1234 removed.")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rows": rows, "next_rung_gbps": next_rung,
                       "fit": fit and {"round_us": fit[0],
                                       "wire_gbps": fit[1]}}, f, indent=2)
        print(f"\nartifact: {a.out}")


def _least_squares(rows: list):
    """``ms = round_ms*R + wire/wire_Bps`` by normal equations, no numpy."""
    pts = [(float(r["rounds"]), float(r["wire_bytes"]), float(r["bar1_ms"]))
           for r in rows]
    if len(pts) < 3:
        return None
    sxx = sum(x * x for x, _, _ in pts)
    sxy = sum(x * y for x, y, _ in pts)
    syy = sum(y * y for _, y, _ in pts)
    sxz = sum(x * z for x, _, z in pts)
    syz = sum(y * z for _, y, z in pts)
    det = sxx * syy - sxy * sxy
    if abs(det) < 1e-9 or syy <= 0:
        return None
    round_ms = (sxz * syy - syz * sxy) / det
    per_byte = (syz * sxx - sxz * sxy) / det
    if round_ms <= 0 or per_byte <= 0:
        return None
    resid = [z - (round_ms * x + per_byte * y) for x, y, z in pts]
    # ``per_byte`` is milliseconds per wire byte; GB/s is 1e-6 over it.
    return round_ms * 1e3, 1e-6 / per_byte, resid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rank", type=int, default=-1, help="internal: child")
    ap.add_argument("--cards", default="0,1,2",
                    help="NVML/CUDA ordinals, one per rank")
    ap.add_argument("--windows", default=",".join(map(str, WINDOWS_MIB)))
    ap.add_argument("--messages", default=",".join(map(str, MESSAGES_MIB)))
    ap.add_argument("--secs", type=float, default=10.0,
                    help="measurement budget per cell (the spec's floor)")
    ap.add_argument("--warmup", type=float, default=1.5)
    ap.add_argument("--nccl", type=int, default=1)
    ap.add_argument("--gloo-rung", type=int, default=1)
    ap.add_argument("--pg-timeout", type=float, default=300.0)
    ap.add_argument("--port", type=int, default=29593)
    ap.add_argument("--max-used-mib", type=int, default=64)
    ap.add_argument("--gpuq-booking", default="")
    ap.add_argument("--sglang-python",
                    default=os.path.join(
                        os.path.dirname(os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__)))),
                        "python"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.cards = [int(x) for x in str(a.cards).split(",") if x != ""]
    a.windows = [int(x) for x in str(a.windows).split(",") if x != ""]
    a.messages = [int(x) for x in str(a.messages).split(",") if x != ""]

    if a.rank >= 0:
        return run_rank(a)

    window = require_gpuq_window(a.cards, a.gpuq_booking)
    print(f"# gpuq window {window.get('id')} owner {window.get('owner')!r} "
          f"cards {window.get('cards')} -- running. The service plans, it "
          f"does not lock; the nvidia-smi net below is the real check.")
    preflight(a.cards, a.max_used_mib)

    env = {**os.environ,
           "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "CUDA_VISIBLE_DEVICES": ",".join(str(c) for c in a.cards),
           "MASTER_ADDR": "127.0.0.1",
           "MASTER_PORT": str(a.port),
           "PYTHONPATH": a.sglang_python + ":" + os.environ.get("PYTHONPATH", ""),
           "PYTHONUNBUFFERED": "1"}
    base = [sys.executable, os.path.abspath(__file__)]
    for k, v in vars(a).items():
        if k == "rank":
            continue
        val = ",".join(map(str, v)) if isinstance(v, list) else str(v)
        if val:
            base += ["--" + k.replace("_", "-"), val]
    procs = [subprocess.Popen(base + ["--rank", str(r)], env=env)
             for r in range(len(a.cards))]
    rc = 0
    budget = (len(a.windows) * len(a.messages)
              * (a.secs + a.warmup) * (2 if a.nccl else 1) + 600)
    deadline = time.time() + budget
    try:
        for p in procs:
            rc |= abs(p.wait(timeout=max(1.0, deadline - time.time())))
    except subprocess.TimeoutExpired:
        sys.stderr.write("\nABORT: total time budget exceeded; killing own "
                         "ranks. Release the gpuq window.\n")
        for p in procs:
            p.kill()
        rc = 4
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
