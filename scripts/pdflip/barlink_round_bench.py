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
   with a correctness check on the result. Until one has, the widened
   coverage is arithmetic, not a measurement. 128 MiB at a 16-MiB window is
   33 rounds.

WHAT IT HAS ALREADY DELIVERED
=============================
Run 2026-09-07 18:20Z, gpuq window ``jpvycx``, cards 0/1/2, world 3; artifact
``/spinning/gpu-arb/weg2/barlink-0907/roundbench_fixed_0907.json`` (+ ``.out``).
Per-round term in isolation 319.2 / 325.4 us (median 322.3), joint fit
323.2 us and 6.02 GB/s with residual mean 0.026 ms / max 0.052 ms over four
rows, and **25- and 33-round decompositions executed with CORRECT results** --
the first above 16 rounds on this rig. Those were the numbers
``barlink_bar1.DEFAULT_{ROUND_US,WIRE_GBPS}`` carried until 23.10.

Run 2026-10-23 09:10Z, gpuq window ``wxs2e5``, cards 0 (5090) + 1 (3080),
world 2, PEX880xx switch-direct (ACS 0011); artifact
``/spinning/gpu-arb/weg2/barlink-1023/p2p_rebar_1023_0909.B_switch_direct.json``:
155.5 us and 20.45 GB/s, residual max 0.064 ms over six rows, a 19-round
decomposition CORRECT -- what ``DEFAULT_*`` carry since barlink-switch-1023.
The 3-card runs of the same morning were contended (card 2 under a foreign
desktop load) and non-linear; since then the report refuses such a run as a
calibration (``contention_verdict`` / ``calibration_verdict``).

The first version of this script could not have delivered any of it: its
correctness gate called the out-of-place ``comm.all_reduce`` for effect and
asserted on the untouched input, so it aborted with ``WRONG RESULT`` at every
window -- including window 40 / 10 rounds, the operating point boot
``weg2bl1`` served 30 100 tokens through correctly. That control is what
identified the bench, and not bar1, as the broken party. ``verify_all_reduce``
is now the single gate, and ``test/registered/unit/pdflip/
test_barlink_round_bench_1234.py`` exercises it against an out-of-place, an
in-place and a deliberately wrong communicator without any hardware.

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

    /opt/gpuq/bin/gpuq book --owner pdflip-barlink --cards 0,1,2 \\
        --duration 20m --purpose "#1234 round-term microbench"

    scripts/pdflip/barlink_round_bench.py --out /spinning/gpu-arb/weg2/roundbench.json

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
        f"  /opt/gpuq/bin/gpuq book --owner pdflip-barlink --cards "
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
# contention and fit quality: when a run is NOT a calibration (23.10.)
# ---------------------------------------------------------------------------
#
# barlink-switch-1023: the two 3-card runs of 23.10. (09:14Z host detour,
# 09:21Z switch-direct) gave w16 96 MiB = 47.48 ms on BOTH topologies and a
# joint fit of 1320 us / 9.8 GB/s and 1816 us / 172 GB/s with residuals up to
# 3.0 ms -- while card 2 carried the user's desktop load (NVML 24-41 %,
# 87-96 W, 2.4 GB in every 60-s sample 09:12-09:22Z). Every round of a
# barlink collective is one kernel launch per rank that spins until all
# ranks arrived; on a time-sliced card the round waits for the foreign
# context's slice, which no PCIe path changes. preflight() only looked at
# memory.used. The run now samples the cards before it starts (and after it
# ends) and the report refuses to call a contended or non-linear fit a
# calibration -- the numbers stay in the artifact, the sentence that hands
# them to barlink_bar1.DEFAULT_* does not appear.

#: Utilization a card may show with none of our processes on it before the
#: run counts as contended. The idle cards of 23.10. read 0 % in every sample.
CONTENDED_UTIL_PCT = 5.0
#: A joint fit whose worst residual exceeds max(this, 5 % of the shortest
#: measured cell) is not a two-term law. The accepted runs: 0.052 ms (0907),
#: 0.064 ms (switch 0909), 0.096 ms (host 0909); the refused 3-card runs
#: 2.06 and 3.01 ms.
FIT_RESIDUAL_MAX_MS = 0.25


def contention_sample(cards: list, samples: int = 4, gap_s: float = 0.5,
                      run=subprocess.run) -> dict:
    """``{card: {"util_max", "power_max", "mem_used_mib"}}`` over a few NVML
    reads (utilization is a moving window; one read can miss a burst)."""
    want = {int(c) for c in cards}
    out = {c: {"util_max": 0.0, "power_max": 0.0, "mem_used_mib": 0}
           for c in want}
    for i in range(max(1, int(samples))):
        try:
            txt = run(["nvidia-smi",
                       "--query-gpu=index,utilization.gpu,power.draw,memory.used",
                       "--format=csv,noheader,nounits"],
                      capture_output=True, text=True, check=True).stdout
        except Exception as e:  # noqa: BLE001 -- recorded, never fatal
            for c in want:
                out[c]["error"] = f"{type(e).__name__}: {e}"
            break
        for line in txt.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            if idx not in want:
                continue
            row = out[idx]

            def _f(x):
                try:
                    return float(x)
                except ValueError:
                    return 0.0
            row["util_max"] = max(row["util_max"], _f(parts[1]))
            row["power_max"] = max(row["power_max"], _f(parts[2]))
            row["mem_used_mib"] = max(row["mem_used_mib"], int(_f(parts[3])))
        if i + 1 < samples:
            time.sleep(gap_s)
    return out


def contention_verdict(*snapshots, util_pct: float = CONTENDED_UTIL_PCT) -> list:
    """Named reasons (empty = clean) from snapshots taken while NONE of our
    processes ran on the cards -- every percent there is somebody else's."""
    why = []
    for snap in snapshots:
        for card, row in sorted((snap or {}).items(), key=lambda kv: int(kv[0])):
            if float(row.get("util_max", 0.0)) > util_pct:
                why.append(f"card {card}: {row['util_max']:.0f} % utilization, "
                           f"{row.get('power_max', 0):.0f} W, "
                           f"{row.get('mem_used_mib', 0)} MiB used with none "
                           f"of this bench's processes on it")
    return why


def calibration_verdict(rows: list, fit, contended: list) -> str:
    """'' when this run may feed barlink_bar1.DEFAULT_*, else why not."""
    if contended:
        return "CONTENDED -- " + "; ".join(contended)
    if not fit:
        return "no joint fit (fewer than 3 independent cells)"
    resid = fit[2]
    shortest = min(float(r["bar1_ms"]) for r in rows) if rows else 0.0
    limit = max(FIT_RESIDUAL_MAX_MS, 0.05 * shortest)
    worst = max(abs(x) for x in resid)
    if worst > limit:
        return (f"NONLINEAR -- worst residual {worst:.3f} ms > {limit:.3f} ms: "
                f"time is not round_us*R + wire/wire_Bps here")
    if not all(r.get("correct") for r in rows):
        return "a cell returned a WRONG result"
    return ""


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


def reduced_result(comm, probe):
    """The tensor that carries the result, whichever contract ``comm`` has.

    ``BarlinkCommunicator.all_reduce`` is OUT-OF-PLACE -- "Returns a new
    tensor", ``barlink.py:1114-1119`` -- while ``torch.distributed.all_reduce``
    mutates and returns ``None``. Calling the first for effect and then
    reading the input back is how this script's correctness gate came to fail
    at 25 rounds while the very same geometry served 30 100 tokens correctly
    in boot ``weg2bl1``: every rank read back its own ``rank + 1``, and a
    broken verification reported a transport defect that was not there.
    """
    out = comm.all_reduce(probe)
    return probe if out is None else out


def verify_all_reduce(comm, buf, world: int, where: str, want=None):
    """One all_reduce of ``buf``, checked. Returns the reduced tensor.

    Runs on a CLONE, so the caller's buffer is reusable whichever contract
    the communicator has -- the timing loop below depends on that. Raises
    ``SystemExit`` on a wrong sum: a decomposition that returns the wrong
    answer is not a slow case, and no timing taken after it means anything.

    ``want`` is the expected tensor (a :func:`round_payload` pair); ``None``
    keeps the old scalar ``sum(1..world)`` for a constant ``rank + 1`` fill.
    """
    probe = reduced_result(comm, buf.clone())
    import torch

    if want is None:
        want = float(sum(range(1, world + 1)))
        ok = bool(torch.all(probe == want).item())
    else:
        ok = bool(torch.equal(probe, want))
    if not ok:
        bad = (probe != want)
        nbad = int(bad.sum().item())
        first = int(bad.nonzero()[0].item()) if nbad else -1
        st = transport_status(comm) if hasattr(comm, "transport") else ""
        raise SystemExit(
            f"WRONG RESULT: {where} -- {nbad} of {probe.numel()} values wrong "
            f"(first at element {first}), got {probe.min().item()}.."
            f"{probe.max().item()} {st}. A decomposition that returns the wrong "
            f"sum is not a slow case."
        )
    return probe


# ---------------------------------------------------------------------------
# The ROUND TERM (Fix A, 22.10.). A constant ``rank + 1`` fill is blind to a
# receiver that reads a STALE slot: the previous round left the very same
# value there, and the sum comes out right. bar1_allreduce.cu carried a round
# term for exactly that reason ("(d) Beleg"); it was lost in the port
# (barlink_bar1_ext.py docstring, "No round term rt") and with it the only
# instrument that could see switch-p2p-wurzel-1022 §2.4. Every call below
# gets a payload that differs per ROUND (k), per ELEMENT (i) and per RANK, and
# never contains the 0 that an untouched slot holds:
#
#     x[r, k, i] = (r + 1) + 8 * ((7 i + 3 k) mod 61)      (exact in fp32)
#     sum_r      = R(R+1)/2 + 8 R ((7 i + 3 k) mod 61)
# ---------------------------------------------------------------------------

def transport_status(comm) -> str:
    """The BAR1 status word, for a WRONG line: 1 = payload barrier, 2 = entry
    ack wait, 3 = a framed receive never verified a tile (Fix A). Reading it
    synchronizes -- only ever called after a wrong result."""
    try:
        return f"status={int(comm.transport.status())}"
    except Exception as e:      # noqa: BLE001 -- a diagnostic, never the error
        return f"status=? ({type(e).__name__})"


def round_payload(n: int, rank: int, world: int, k: int, device):
    """``(payload, expected all_reduce sum)`` for call ``k``, float32, exact."""
    import torch

    i = torch.arange(n, device=device, dtype=torch.int64)
    pat = ((7 * i + 3 * int(k)) % 61).to(torch.float32) * 8.0
    payload = pat + float(rank + 1)
    want = pat * float(world) + float(world * (world + 1) // 2)
    return payload, want


def gather_payload(n: int, rank: int, k: int, device):
    """Shard of rank ``rank`` for all_gather/broadcast/a2a call ``k``."""
    import torch

    i = torch.arange(n, device=device, dtype=torch.int64)
    return (((7 * i + 3 * int(k) + 11 * rank) % 251) + 1 + 256 * rank).to(
        torch.float32)


def _timed_verified(comm, pairs: list, seconds: float, where: str,
                    at_least: int = 3) -> list:
    """Like :func:`_timed`, rotating over round-termed ``(payload, want)``
    pairs, and every single result checked after its clock stopped."""
    import torch

    samples = []
    wrong = []
    deadline = time.perf_counter() + seconds
    j = 0
    while time.perf_counter() < deadline or len(samples) < at_least:
        buf, want = pairs[j % len(pairs)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = comm.all_reduce(buf)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
        res = buf if out is None else out
        if not bool(torch.equal(res, want)):
            wrong.append((j, int((res != want).sum().item())))
        j += 1
    if wrong:
        raise SystemExit(
            f"WRONG RESULT: {where} -- {len(wrong)} of {j} timed calls wrong "
            f"(call, n_bad) {wrong[:8]} {transport_status(comm)}. A "
            f"decomposition that returns the wrong sum is not a slow case.")
    return samples


def check_other_ops(comm, n: int, rank: int, world: int, dev, kctr: list,
                    where: str, log) -> None:
    """all_gather, broadcast and all_to_all_single with the round term.

    Same transport, same window -- the a2a kernel carries all three, and a
    stale slot shows the same way: a value from an earlier call, or 0.
    Raises SystemExit on any wrong element.
    """
    import torch

    tr = comm.transport
    shard = max(4, n // world // 4 * 4)
    # all_gather
    k = kctr[0]
    kctr[0] += 1
    mine = gather_payload(shard, rank, k, dev)
    got = comm.all_gather(mine, dim=0)
    want = torch.cat([gather_payload(shard, r, k, dev) for r in range(world)])
    path = "bar1" if tr.handles("all_gather", shard * 4) else "host"
    if not torch.equal(got, want):
        raise SystemExit(f"WRONG RESULT: {where} all_gather ({path}) -- "
                         f"{int((got != want).sum())} of {want.numel()} wrong")
    # broadcast from every source
    for src in range(world):
        k = kctr[0]
        kctr[0] += 1
        t = gather_payload(shard, src, k, dev) if rank == src else torch.zeros(
            shard, dtype=torch.float32, device=dev)
        got = comm.broadcast(t, src)
        want = gather_payload(shard, src, k, dev)
        if not torch.equal(got, want):
            raise SystemExit(f"WRONG RESULT: {where} broadcast src={src} -- "
                             f"{int((got != want).sum())} of {shard} wrong")
    # all_to_all_single, even split
    k = kctr[0]
    kctr[0] += 1
    blk = max(4, shard // world // 4 * 4)
    inp = torch.cat([gather_payload(blk, rank * world + z, k, dev)
                     for z in range(world)])
    out = torch.empty_like(inp)
    comm.all_to_all_single(out, inp)
    want = torch.cat([gather_payload(blk, s * world + rank, k, dev)
                      for s in range(world)])
    if not torch.equal(out, want):
        raise SystemExit(f"WRONG RESULT: {where} all_to_all_single -- "
                         f"{int((out != want).sum())} of {want.numel()} wrong")
    log(f"#   {where}: all_gather({path}, {shard * 4} B shard), broadcast x{world}, "
        f"all_to_all_single ({blk * 4} B blocks) CORRECT")


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

    sys.path.insert(0, a.flliper_python)
    import flliper.srt.distributed.device_communicators.barlink as barlink_mod
    from flliper.srt.distributed.device_communicators.barlink_bar1 import (
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
    os.environ["FLLIPER_BARLINK_BAR1_AR_MAX_ROUNDS"] = "100000"
    os.environ["FLLIPER_BARLINK_BAR1_AG_MAX_ROUNDS"] = "100000"
    os.environ["FLLIPER_BARLINK_BAR1_A2A_MAX_ROUNDS"] = "100000"
    os.environ["FLLIPER_BARLINK_BAR1_BC_MAX_ROUNDS"] = "100000"
    os.environ["FLLIPER_BARLINK_UNCOVERED_CLASS"] = "refuse"

    sizes = [m << 20 for m in a.messages]
    ref = None
    kctr = [0]          # the round term's call counter, global over the run

    for wmib in a.windows:
        os.environ["FLLIPER_BARLINK_BAR1_WINDOW_MIB"] = str(wmib)
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
            rounds = len(ar_plan(nbytes, chunk_max, world))
            where = f"window {wmib} MiB, {nbytes} B, {rounds} rounds"

            # -- correctness, BEFORE any timing, WITH THE ROUND TERM. A fast
            # -- wrong answer is the one failure mode a throughput bench
            # -- cannot see; a constant payload cannot see a stale slot.
            for c in range(a.verify_calls):
                buf, want = round_payload(n, rank, world, kctr[0], dev)
                kctr[0] += 1
                verify_all_reduce(comm, buf, world, f"{where}, call {c}",
                                  want=want)
            ok = True

            # -- timing on ROTATING round-termed payloads; every timed result
            # -- is verified OUTSIDE the timed span, so a wrong timed call
            # -- still ends the run.
            pairs = []
            for _ in range(3):
                pairs.append(round_payload(n, rank, world, kctr[0], dev))
                kctr[0] += 1
            _timed_verified(comm, pairs, a.warmup, f"{where} (warmup)")
            bar1 = _timed_verified(comm, pairs, a.secs, f"{where} (timed)")
            buf = pairs[0][0]
            if a.other_ops:
                check_other_ops(comm, n, rank, world, dev, kctr, where, log)

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
    try:
        pre = json.loads(os.environ.get("BARLINK_BENCH_CONTENTION_PRE", "")
                         or "{}")
    except ValueError:
        pre = {}
    contended = contention_verdict(pre)
    refused = calibration_verdict(rows, fit, contended)
    if refused:
        print(f"\nNOT A CALIBRATION: {refused}\nThe rows above stay in the "
              "artifact as a measurement of THIS condition; they must not be "
              "copied into barlink_bar1.DEFAULT_*.")
    else:
        print("\nThese three numbers are what "
              "barlink_bar1.DEFAULT_{ROUND_US,WIRE_GBPS,NEXT_RUNG_GBPS} must "
              "carry, together with this run's artifact path, windows, round "
              "counts and residuals. A bare number in that docstring is the "
              "defect #1234 removed.")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rows": rows, "next_rung_gbps": next_rung,
                       "fit": fit and {"round_us": fit[0],
                                       "wire_gbps": fit[1],
                                       "residuals_ms": list(fit[2])},
                       "contention_pre": pre,
                       "calibration_ok": not refused,
                       "calibration_refused": refused}, f, indent=2)
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
    ap.add_argument("--flliper-python",
                    default=os.path.join(
                        os.path.dirname(os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__)))),
                        "python"))
    ap.add_argument("--out", default="")
    ap.add_argument("--verify-calls", type=int, default=4,
                    help="round-termed, verified all_reduce calls per cell "
                         "before timing")
    ap.add_argument("--other-ops", type=int, default=1,
                    help="also verify all_gather/broadcast/a2a per cell")
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
    pre = contention_sample(a.cards)
    for line in contention_verdict(pre):
        print(f"# CONTENDED before start: {line}")

    env = {**os.environ,
           "BARLINK_BENCH_CONTENTION_PRE": json.dumps(pre),
           "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "CUDA_VISIBLE_DEVICES": ",".join(str(c) for c in a.cards),
           "MASTER_ADDR": "127.0.0.1",
           "MASTER_PORT": str(a.port),
           "PYTHONPATH": a.flliper_python + ":" + os.environ.get("PYTHONPATH", ""),
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
    # After our ranks are gone every percent is foreign again: a load that
    # began DURING the run shows here. Appended to the artifact, and it
    # withdraws the calibration verdict if it finds anything.
    post = contention_sample(a.cards)
    late = contention_verdict(post)
    for line in late:
        print(f"# CONTENDED after the run: {line}")
    if a.out and os.path.exists(a.out):
        try:
            with open(a.out) as f:
                doc = json.load(f)
            doc["contention_post"] = post
            if late and doc.get("calibration_ok"):
                doc["calibration_ok"] = False
                doc["calibration_refused"] = "CONTENDED after the run -- " + "; ".join(late)
                print(f"NOT A CALIBRATION: {doc['calibration_refused']}")
            with open(a.out, "w") as f:
                json.dump(doc, f, indent=2)
        except (OSError, ValueError) as e:
            print(f"# could not append the post-run contention sample: {e!r}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
